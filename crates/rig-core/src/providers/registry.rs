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

use crate::catalog::ModelSpec;
#[cfg(feature = "reqwest")]
use crate::client::env::{self, EnvError};
use crate::completion::ModelRef;
use crate::driver::DynModel;
use crate::http_client::{DynHttpClient, HttpClientExt};
use crate::operation::Completion;
use crate::providers::{anthropic, gemini, openai};
use crate::serve::ErasedHandler;
use crate::serve::adapters::ModelAdapter;
use crate::wire::Secret;

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
    &openai::wire::COHERE,
    &openai::wire::OLLAMA,
    &crate::providers::xai::DIALECT,
    &crate::providers::chatgpt::DIALECT,
    &crate::providers::copilot::wire::DIALECT,
];

/// A provider's request grammar and configuration type.
/// The OpenAI family includes Chat Completions and Responses; select the endpoint
/// through [`openai::OpenAIConfig::with_route`].
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

/// Identity is the `(vendor, format)` pair, as for [`ProviderId`].
impl PartialEq for Registered {
    fn eq(&self, other: &Self) -> bool {
        self.vendor() == other.vendor() && self.format() == other.format()
    }
}

impl Registered {
    fn vendor(&self) -> &'static str {
        match self {
            Self::OpenAi(dialect) => dialect.name,
            Self::Anthropic(dialect) => dialect.name,
            Self::Gemini => gemini::PROVIDER_NAME,
        }
    }

    fn format(&self) -> Format {
        match self {
            Self::OpenAi(_) => Format::OpenAi,
            Self::Anthropic(_) => Format::Anthropic,
            Self::Gemini => Format::Gemini,
        }
    }

    fn config(&self, api_key: impl Into<Secret>) -> ProviderConfig {
        match self {
            Self::OpenAi(dialect) => {
                ProviderConfig::OpenAi(openai::wire::OpenAIConfig::with_key(dialect, api_key))
            }
            Self::Anthropic(dialect) => ProviderConfig::Anthropic(
                anthropic::wire::AnthropicConfig::with_key(dialect, api_key),
            ),
            Self::Gemini => ProviderConfig::Gemini(gemini::GeminiConfig::new(api_key)),
        }
    }

    /// This selection's preset, configured from the environment variables
    /// its dialect names.
    #[cfg(feature = "reqwest")]
    fn config_from_env(&self) -> Result<ProviderConfig, EnvError> {
        Ok(match self {
            Self::OpenAi(dialect) => {
                let (api_key, auth) = openai_credential_from_env(dialect)?;
                ProviderConfig::OpenAi(openai::wire::OpenAIConfig::from_env_with_credential(
                    dialect, api_key, auth,
                )?)
            }
            Self::Anthropic(dialect) => {
                ProviderConfig::Anthropic(anthropic::wire::AnthropicConfig::from_env_with(dialect)?)
            }
            Self::Gemini => ProviderConfig::Gemini(gemini::GeminiConfig::from_env()?),
        })
    }
}

/// A provider the model catalog files models under that the registry cannot
/// configure: an SDK-backed or local provider in a companion crate, or one
/// that serves no completions.
#[derive(Debug)]
struct CatalogOnly {
    /// The provider's own descriptor name.
    vendor: &'static str,
    /// Where its models are served from, for the error `connect` returns.
    home: &'static str,
    /// Whether it needs a credential.
    credential: bool,
}

/// Every catalog-only provider.
static CATALOG_ONLY: [CatalogOnly; 5] = [
    CatalogOnly {
        vendor: "aws_bedrock",
        home: "the `rig-bedrock` crate",
        credential: true,
    },
    CatalogOnly {
        vendor: "vertexai",
        home: "the `rig-vertexai` crate",
        credential: true,
    },
    CatalogOnly {
        vendor: "gemini-grpc",
        home: "the `rig-gemini-grpc` crate",
        credential: true,
    },
    CatalogOnly {
        vendor: "candle",
        home: "the `rig-candle` crate",
        credential: false,
    },
    CatalogOnly {
        vendor: "voyageai",
        home: "`rig_core::providers::voyageai`, which serves embeddings and reranking only",
        credential: true,
    },
];

/// What a [`ProviderId`] names.
#[derive(Debug, Clone, Copy)]
enum Kind {
    /// A selection the registry configures.
    Registered(Registered),
    /// A provider only the model catalog names.
    CatalogOnly(&'static CatalogOnly),
}

/// Validated vendor and protocol-family pair from this build's dialect tables,
/// or a catalog-only provider the registry cannot configure. Construction and
/// deserialization reject unregistered pairs; only
/// [`ProviderId::catalog`] and the model catalog produce a catalog-only id,
/// which has no format and no preset. Equality and hashing use the pair, not
/// dialect options; display emits `vendor/format`, or the vendor alone for a
/// catalog-only id.
#[derive(Debug, Clone, Copy)]
pub struct ProviderId(Kind);

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
    /// one per Messages-format dialect, and Gemini. Catalog-only providers
    /// are not selections.
    pub fn all() -> impl Iterator<Item = ProviderId> {
        openai::wire::all()
            .map(|dialect| Registered::OpenAi(*dialect))
            .chain(anthropic::wire::all().map(|dialect| Registered::Anthropic(*dialect)))
            .chain(std::iter::once(Registered::Gemini))
            .map(|registered| ProviderId(Kind::Registered(registered)))
    }

    /// The selection `vendor` names in `format`, or `None` when this build
    /// registers no such pair.
    pub fn new(vendor: &str, format: Format) -> Option<Self> {
        let registered = match format {
            Format::OpenAi => {
                openai::wire::by_name(vendor).map(|dialect| Registered::OpenAi(*dialect))
            }
            Format::Anthropic => {
                anthropic::wire::Dialect::by_name(vendor).map(Registered::Anthropic)
            }
            Format::Gemini => (vendor == gemini::PROVIDER_NAME).then_some(Registered::Gemini),
        };
        registered.map(|registered| Self(Kind::Registered(registered)))
    }

    /// The id the model catalog files `vendor`'s models under: its first
    /// registered selection, or a catalog-only id for a provider the
    /// registry cannot configure (`aws_bedrock`, `vertexai`, `gemini-grpc`,
    /// `candle`, `voyageai`). `None` for a vendor this build does not know.
    pub fn catalog(vendor: &str) -> Option<Self> {
        Self::vendor_selections(vendor).next().or_else(|| {
            CATALOG_ONLY
                .iter()
                .find(|provider| provider.vendor == vendor)
                .map(|provider| Self(Kind::CatalogOnly(provider)))
        })
    }

    /// The vendor, spelled as the provider's own descriptor name.
    pub fn vendor(&self) -> &'static str {
        match &self.0 {
            Kind::Registered(registered) => registered.vendor(),
            Kind::CatalogOnly(provider) => provider.vendor,
        }
    }

    /// The protocol family, or `None` for a catalog-only provider.
    pub fn format(&self) -> Option<Format> {
        match &self.0 {
            Kind::Registered(registered) => Some(registered.format()),
            Kind::CatalogOnly(_) => None,
        }
    }

    /// Whether the registry can configure this provider. A catalog-only
    /// provider is served by its companion crate instead.
    pub fn is_registered(&self) -> bool {
        matches!(self.0, Kind::Registered(_))
    }

    /// Where a catalog-only provider's models are served from, or `None`
    /// for a registered selection.
    pub(crate) fn served_by(&self) -> Option<&'static str> {
        match &self.0 {
            Kind::Registered(_) => None,
            Kind::CatalogOnly(provider) => Some(provider.home),
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
    /// each of which this resolver accepts. A catalog-only provider is not a
    /// selection, so it does not resolve.
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

    /// Build this selection's preset with `api_key`, or `None` for a
    /// catalog-only provider. Copilot requires an exchanged session token,
    /// not a GitHub OAuth token.
    pub fn config(&self, api_key: impl Into<Secret>) -> Option<ProviderConfig> {
        match &self.0 {
            Kind::Registered(registered) => Some(registered.config(api_key)),
            Kind::CatalogOnly(_) => None,
        }
    }

    /// Environment variable named by the registered credential configuration,
    /// or `None` for a catalog-only provider, whose companion crate reads its
    /// own credentials.
    pub fn api_key_env(&self) -> Option<&'static str> {
        match &self.0 {
            Kind::Registered(Registered::OpenAi(dialect)) => Some(dialect.api_key_env),
            Kind::Registered(Registered::Anthropic(dialect)) => Some(dialect.api_key_env),
            Kind::Registered(Registered::Gemini) => Some(gemini::API_KEY_ENV),
            Kind::CatalogOnly(_) => None,
        }
    }

    /// Whether this selection needs a credential at all.
    ///
    /// A local `llama-server` authenticates optionally, so naming its
    /// variable is a hint rather than a requirement.
    pub fn requires_credential(&self) -> bool {
        match &self.0 {
            Kind::Registered(Registered::OpenAi(dialect)) => {
                !matches!(dialect.quirks.auth, openai::wire::Auth::OptionalBearer)
            }
            Kind::Registered(Registered::Anthropic(_) | Registered::Gemini) => true,
            Kind::CatalogOnly(provider) => provider.credential,
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
        match self.format() {
            Some(format) => write!(f, "{}/{format}", self.vendor()),
            None => f.write_str(self.vendor()),
        }
    }
}

impl Serialize for ProviderId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// Reads a selection as [`ProviderId::resolve`] does, and a catalog-only
/// provider by its vendor name, so a catalog id reads back.
impl<'de> Deserialize<'de> for ProviderId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let selection = String::deserialize(deserializer)?;
        Self::resolve(&selection)
            .or_else(|error| {
                Self::catalog(&selection)
                    .filter(|id| !id.is_registered())
                    .ok_or(error)
            })
            .map_err(de::Error::custom)
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
    OpenAi(openai::wire::OpenAIConfig),
    /// A Messages-format provider.
    #[serde(rename = "anthropic")]
    Anthropic(anthropic::wire::AnthropicConfig),
    /// Gemini.
    #[serde(rename = "gemini")]
    Gemini(gemini::GeminiConfig),
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

    /// The base URL this configuration sends requests to. Endpoint paths,
    /// such as `/chat/completions` or `/v1/messages`, resolve against it.
    /// A route the dialect serves at the server root drops a trailing `/v1`
    /// from it first.
    ///
    /// A Copilot preset built with a session token holds the endpoint that
    /// token names. [`with_credential`](Self::with_credential) keeps the URL
    /// it finds.
    pub fn base_url(&self) -> &str {
        match self {
            Self::OpenAi(provider) => &provider.base_url,
            Self::Anthropic(provider) => &provider.base_url,
            Self::Gemini(provider) => &provider.base_url,
        }
    }

    /// The same configuration sending to `base_url`, such as a proxy, with
    /// every other setting kept. A Messages-format configuration normalizes
    /// the URL with [`anthropic::wire::normalize_base_url`].
    pub fn with_base_url(self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        match self {
            Self::OpenAi(provider) => Self::OpenAi(provider.with_base_url(base_url)),
            Self::Anthropic(provider) => Self::Anthropic(provider.with_base_url(base_url)),
            Self::Gemini(provider) => Self::Gemini(provider.with_base_url(base_url)),
        }
    }

    /// The completion wire for `model`, bound to `http`, erased behind a
    /// [`ModelAdapter`] labelled `label`.
    ///
    /// A handler built from data uses the provider's own completion wire,
    /// including Copilot's model-dependent routing and request envelope.
    /// Explicit hosts, routes and typed options are preserved.
    pub fn completion_handler(
        &self,
        label: &str,
        model: &str,
        http: DynHttpClient,
    ) -> ErasedHandler {
        ErasedHandler::new(ModelAdapter::new(label, self.completion_model(model, http)))
    }

    /// The provider's completion model for `model` on `http`, erased.
    fn completion_model(&self, model: &str, http: DynHttpClient) -> DynModel<Completion> {
        match self {
            Self::OpenAi(provider) => provider.clone().connect(http).completion(model).erase(),
            Self::Anthropic(provider) => provider.clone().connect(http).completion(model).erase(),
            Self::Gemini(provider) => provider.clone().connect(http).completion(model).erase(),
        }
    }

    /// The credential this configuration's vendor reads from the
    /// environment. A vendor whose credential is optional reads an unset
    /// variable as no credential.
    #[cfg(feature = "reqwest")]
    fn with_credential_from_env(self) -> Result<Self, EnvError> {
        Ok(match self {
            Self::OpenAi(mut provider) => {
                let (api_key, auth) = openai_credential_from_env(&provider.dialect)?;
                provider.api_key = api_key.into();
                // Only an alternative credential changes how it is sent.
                if auth != provider.dialect.quirks.auth {
                    provider.auth = auth;
                }
                Self::OpenAi(provider)
            }
            Self::Anthropic(mut provider) => {
                provider.api_key = env::required(provider.dialect.api_key_env)?.into();
                Self::Anthropic(provider)
            }
            Self::Gemini(mut provider) => {
                provider.api_key = env::required(gemini::API_KEY_ENV)?.into();
                Self::Gemini(provider)
            }
        })
    }
}

/// The credential an OpenAI-family `dialect` reads from the environment:
/// its alternative when only that is set, and none when the dialect
/// authenticates optionally and nothing is set.
#[cfg(feature = "reqwest")]
fn openai_credential_from_env(
    dialect: &openai::wire::Dialect,
) -> Result<(String, openai::wire::Auth), EnvError> {
    if matches!(dialect.quirks.auth, openai::wire::Auth::OptionalBearer)
        && dialect.alternate_auth.is_none()
    {
        let api_key = env::optional(dialect.api_key_env)?.unwrap_or_default();
        return Ok((api_key, dialect.quirks.auth));
    }
    openai::wire::OpenAIConfig::credential_from_env(dialect)
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
    recipe: Recipe,
    /// The provider's own model identifier. Non-empty: the string form
    /// separates the selection from the model at the first `:`, so an empty
    /// model has no spelling [`parse`](Self::parse) would read back.
    model: String,
}

/// What a [`ProviderRef`] builds from: a registered preset, never a
/// catalog-only provider, or an explicit configuration.
#[derive(Clone, Debug, PartialEq)]
enum Recipe {
    Registered(Registered),
    Configured(ProviderConfig),
}

impl ProviderRef {
    /// A reference to a non-empty `model` on a registered selection.
    /// Returns [`RefError::EmptyModel`] for an empty identifier, and
    /// [`SelectionError::Unknown`] for a catalog-only provider, which the
    /// registry cannot configure.
    pub fn registered(id: ProviderId, model: impl Into<String>) -> Result<Self, RefError> {
        match id.0 {
            Kind::Registered(registered) => Self::new(Recipe::Registered(registered), model.into()),
            Kind::CatalogOnly(provider) => Err(RefError::Selection(SelectionError::Unknown {
                vendor: provider.vendor.to_owned(),
            })),
        }
    }

    /// A reference to a non-empty `model` on an explicit configuration.
    /// Removes the credential: references are persistent recipes, and the
    /// host supplies credentials through [`config`](Self::config) at construction time.
    /// The host stays fixed, even if it is a preset's default. For Copilot,
    /// use a registered reference to follow the resolved token's proxy endpoint,
    /// or configure the intended endpoint before creating this reference.
    /// Returns [`RefError::EmptyModel`] for an empty identifier.
    pub fn configured(config: ProviderConfig, model: impl Into<String>) -> Result<Self, RefError> {
        Self::new(Recipe::Configured(config.with_credential("")), model.into())
    }

    /// The credential-free provider recipe.
    pub fn provider(&self) -> Provider {
        match &self.recipe {
            Recipe::Registered(registered) => {
                Provider::Registered(ProviderId(Kind::Registered(*registered)))
            }
            Recipe::Configured(config) => Provider::Configured(config.clone()),
        }
    }

    fn new(recipe: Recipe, model: String) -> Result<Self, RefError> {
        if model.is_empty() {
            return Err(RefError::EmptyModel);
        }
        Ok(Self { recipe, model })
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
        match &self.recipe {
            Recipe::Registered(registered) => Some(ProviderId(Kind::Registered(*registered))),
            Recipe::Configured(config) => config.id(),
        }
    }

    /// The configuration to build from, credentialed with `api_key`: the
    /// registry's preset, or the explicit configuration rehydrated.
    pub fn config(&self, api_key: impl Into<Secret>) -> ProviderConfig {
        match &self.recipe {
            Recipe::Registered(registered) => registered.config(api_key),
            Recipe::Configured(config) => config.clone().with_credential(api_key),
        }
    }
}

impl ProviderRef {
    /// A completion model for this reference, credentials from the
    /// environment, on the shared reqwest client. A registered selection
    /// reads every variable its dialect names; an explicit configuration
    /// keeps its host and options and reads only its vendor's credential.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn completion_model(&self) -> Result<DynModel<Completion>, EnvError> {
        let config = match &self.recipe {
            Recipe::Registered(registered) => registered.config_from_env()?,
            Recipe::Configured(config) => config.clone().with_credential_from_env()?,
        };
        Ok(config.completion_model(&self.model, rig_reqwest::shared()))
    }

    /// A completion model for this reference, credentialed with `api_key`,
    /// sending through `http`.
    pub fn completion_model_with(
        &self,
        api_key: impl Into<Secret>,
        http: impl HttpClientExt + 'static,
    ) -> DynModel<Completion> {
        self.config(api_key)
            .completion_model(&self.model, DynHttpClient::new(http))
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
        match &self.recipe {
            Recipe::Registered(registered) => {
                write!(
                    f,
                    "{}/{}:{}",
                    registered.vendor(),
                    registered.format(),
                    self.model
                )
            }
            Recipe::Configured(config) => {
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

/// What [`connect`] connects to: a catalog entry, or a reference spelled as
/// [`Catalog::resolve`](crate::catalog::Catalog::resolve) reads one
/// (`anthropic/claude-opus-5-5`, `deepseek:deepseek-chat`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ModelSelector<'a> {
    /// A catalog entry: its provider and id.
    Spec(&'a ModelSpec),
    /// `vendor/model` or `vendor[/format]:model`. The first form names a
    /// vendor served over two formats by its first registered selection.
    Reference(&'a str),
}

impl<'a> From<&'a ModelSpec> for ModelSelector<'a> {
    fn from(spec: &'a ModelSpec) -> Self {
        Self::Spec(spec)
    }
}

impl<'a> From<&'a str> for ModelSelector<'a> {
    fn from(reference: &'a str) -> Self {
        Self::Reference(reference)
    }
}

impl<'a> From<&'a String> for ModelSelector<'a> {
    fn from(reference: &'a String) -> Self {
        Self::Reference(reference)
    }
}

impl<'a> From<&'a ModelRef> for ModelSelector<'a> {
    fn from(reference: &'a ModelRef) -> Self {
        Self::Reference(reference.as_str())
    }
}

impl ModelSelector<'_> {
    /// The registered reference this selects. A catalog-only provider is
    /// [`ConnectError::CatalogOnly`].
    pub fn provider_ref(self) -> Result<ProviderRef, ConnectError> {
        let (id, model) = match self {
            Self::Spec(spec) => (spec.provider, spec.id.as_str()),
            Self::Reference(reference) => {
                let (vendor, model) =
                    crate::catalog::split_reference(reference).ok_or_else(|| {
                        ConnectError::Malformed {
                            reference: reference.to_owned(),
                        }
                    })?;
                if let Some(id) = ProviderId::catalog(vendor).filter(|id| !id.is_registered()) {
                    (id, model)
                } else if selection_grammar(reference) {
                    return Ok(ProviderRef::parse(reference)?);
                } else {
                    let id = ProviderId::catalog(vendor).ok_or_else(|| {
                        RefError::Selection(SelectionError::Unknown {
                            vendor: vendor.to_owned(),
                        })
                    })?;
                    (id, model)
                }
            }
        };
        match id.served_by() {
            Some(served_by) => Err(ConnectError::CatalogOnly {
                vendor: id.vendor().to_owned(),
                served_by,
            }),
            None => Ok(ProviderRef::registered(id, model)?),
        }
    }
}

/// Whether `reference` is `vendor[/format]:model` rather than `vendor/model`.
fn selection_grammar(reference: &str) -> bool {
    reference.split_once(':').is_some_and(|(selection, _)| {
        selection
            .split_once('/')
            .is_none_or(|(_, format)| Format::named(format).is_some())
    })
}

/// Why [`connect`] built no model.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConnectError {
    /// The reference names no model.
    #[error("`{reference}` names no model: expected `vendor/model` or `vendor[/format]:model`")]
    Malformed {
        /// What was given.
        reference: String,
    },
    /// The provider is one only the catalog knows; its companion crate
    /// serves its models.
    #[error("the registry cannot connect to `{vendor}`: its models are served by {served_by}")]
    CatalogOnly {
        /// The provider's vendor name.
        vendor: String,
        /// Where its models are served from.
        served_by: &'static str,
    },
    /// The reference did not resolve to a registered selection.
    #[error(transparent)]
    Reference(#[from] RefError),
}

/// The completion model `model` selects, credentialed with `api_key`, on
/// the shared reqwest client: the same model
/// [`ProviderRef::completion_model_with`] builds.
///
/// ```no_run
/// use rig_core::catalog::Catalog;
/// use rig_core::providers::registry::connect;
///
/// let model = connect("anthropic/claude-opus-5-5", "sk-ant-...")?;
/// let spec = Catalog::builtin().resolve("openai/gpt-5.5")?.spec;
/// let other = connect(spec, "sk-...")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
pub fn connect<'a>(
    model: impl Into<ModelSelector<'a>>,
    api_key: impl Into<Secret>,
) -> Result<DynModel<Completion>, ConnectError> {
    let reference = model.into().provider_ref()?;
    Ok(reference
        .config(api_key)
        .completion_model(reference.model(), rig_reqwest::shared()))
}

/// The completion model `model` selects, credentialed with `api_key`,
/// sending through `http`.
pub fn connect_with<'a>(
    model: impl Into<ModelSelector<'a>>,
    api_key: impl Into<Secret>,
    http: impl HttpClientExt + 'static,
) -> Result<DynModel<Completion>, ConnectError> {
    Ok(model
        .into()
        .provider_ref()?
        .completion_model_with(api_key, http))
}

/// The field names of the object form, which is also what a wrong shape is
/// reported against.
const REF_FIELDS: &[&str] = &["config", "model"];

/// Serialize registered references as canonical strings and configured references
/// as `{config, model}` objects to preserve their hosts and options.
impl Serialize for ProviderRef {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.recipe {
            Recipe::Registered(_) => serializer.collect_str(self),
            Recipe::Configured(config) => {
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
