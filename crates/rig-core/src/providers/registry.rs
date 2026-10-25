//! Serializable provider selections, explicit configurations, and
//! connecting to a model by reference.
//!
//! A model reference is `vendor/model`, split at the first `/`
//! (`anthropic/claude-opus-5-5`, `ollama/qwen3:4b`). The protocol family it
//! is reached by comes from the catalog row, and [`ConnectOptions`] can name
//! another. [`Catalog::connect`] builds a model from a reference with the key
//! from the environment; a [`ProviderConfig`] is the explicit, serializable
//! configuration, which discards its credential when written.
//!
//! ```no_run
//! use rig_core::catalog::Catalog;
//! let model = Catalog::builtin().connect("deepseek/deepseek-chat")?; // DEEPSEEK_API_KEY
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};

use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::catalog::{Catalog, ModelFacts, ModelSpec, NotFound, Reference};
use crate::client::env::{self, EnvError};
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
    &openai::wire::OPENCODE,
    &openai::wire::OPENCODE_GO,
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
    /// its dialect names: the key from the first of
    /// [`key_envs`](Self::key_envs) set to a non-empty value, and the base
    /// URL and other settings the dialect reads.
    fn config_from_env(&self) -> Result<ProviderConfig, ConnectError> {
        let tried = self.key_envs();
        let (api_key, name) = key_from_env(self.vendor(), tried, self.requires_credential())?;
        Ok(match self {
            Self::OpenAi(dialect) => {
                let auth = openai_auth(dialect, name);
                ProviderConfig::OpenAi(openai::wire::OpenAIConfig::from_env_with_credential(
                    dialect, api_key, auth,
                )?)
            }
            Self::Anthropic(dialect) => ProviderConfig::Anthropic(
                anthropic::wire::AnthropicConfig::from_env_with_credential(dialect, api_key)?,
            ),
            Self::Gemini => ProviderConfig::Gemini(gemini::GeminiConfig::new(api_key)),
        })
    }

    /// The variables this selection reads its key from, in order.
    fn key_envs(&self) -> Vec<&'static str> {
        match self {
            Self::OpenAi(dialect) => openai_key_envs(dialect),
            Self::Anthropic(dialect) => vec![dialect.api_key_env],
            Self::Gemini => vec![gemini::API_KEY_ENV],
        }
    }

    /// Whether this selection's key expires within the hour.
    fn short_lived_key(&self) -> bool {
        matches!(self, Self::OpenAi(dialect) if dialect.quirks.short_lived_key)
    }

    /// Whether this selection needs a credential at all.
    fn requires_credential(&self) -> bool {
        match self {
            Self::OpenAi(dialect) => openai_requires_credential(dialect),
            Self::Anthropic(dialect) => {
                dialect.quirks.auth != anthropic::wire::Auth::OptionalBearer
            }
            Self::Gemini => true,
        }
    }
}

/// The variables an OpenAI-family `dialect` reads its key from: its own,
/// then its alternative's.
fn openai_key_envs(dialect: &openai::wire::Dialect) -> Vec<&'static str> {
    std::iter::once(dialect.api_key_env)
        .chain(
            dialect
                .alternate_auth
                .map(|alternative| alternative.api_key_env),
        )
        .collect()
}

/// Whether an OpenAI-family `dialect` needs a credential: a local
/// `llama-server` or Ollama daemon authenticates optionally.
fn openai_requires_credential(dialect: &openai::wire::Dialect) -> bool {
    !matches!(dialect.quirks.auth, openai::wire::Auth::OptionalBearer)
}

/// How an OpenAI-family `dialect` sends a key read from the variable
/// `name`: its alternative's way for the alternative's variable, its own
/// otherwise.
fn openai_auth(dialect: &openai::wire::Dialect, name: Option<&'static str>) -> openai::wire::Auth {
    match dialect.alternate_auth {
        Some(alternative) if name == Some(alternative.api_key_env) => alternative.auth,
        _ => dialect.quirks.auth,
    }
}

/// The key the first of `tried` holds, and its name. An empty value counts
/// as unset. When none is set, a vendor that needs no credential gets an
/// empty key, and any other is [`ConnectError::MissingKey`].
fn key_from_env(
    vendor: &str,
    tried: Vec<&'static str>,
    required: bool,
) -> Result<(String, Option<&'static str>), ConnectError> {
    for name in &tried {
        if let Some(value) = env::optional(name)?.filter(|value| !value.is_empty()) {
            return Ok((value, Some(*name)));
        }
    }
    if required {
        return Err(ConnectError::MissingKey {
            vendor: vendor.to_owned(),
            tried: tried.into_iter().map(Cow::Borrowed).collect(),
        });
    }
    Ok((String::new(), None))
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
    /// A vendor declared at run time ([`ProviderId::declare`]).
    Declared(&'static str),
}

/// The vendors declared at run time, each name stored once for the life of
/// the process, so a declared [`ProviderId`] stays `Copy` and names its
/// vendor as `&'static str` like a built-in one.
static DECLARED: std::sync::RwLock<Vec<&'static str>> = std::sync::RwLock::new(Vec::new());

/// The declared vendor named `vendor`, if any.
fn declared(vendor: &str) -> Option<&'static str> {
    DECLARED
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .copied()
        .find(|name| *name == vendor)
}

/// Validated vendor and protocol-family pair from this build's dialect tables,
/// a catalog-only provider the registry cannot configure, or a vendor
/// declared at run time ([`Self::declare`]). Construction and
/// deserialization reject unregistered pairs; only [`ProviderId::catalog`],
/// [`ProviderId::declare`] and the model catalog produce a catalog-only or
/// declared id, which has no format and no preset. Equality and hashing use
/// the pair, not dialect options; display emits `vendor/format`, or the
/// vendor alone for a catalog-only or declared id.
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

    /// The id the model catalog files `vendor`'s models under: its
    /// primary selection, or a catalog-only id for a provider the registry
    /// cannot configure (`aws_bedrock`, `vertexai`, `gemini-grpc`, `candle`,
    /// `voyageai`). `None` for a vendor this build does not know.
    ///
    /// A vendor registered in one family has that family as its primary.
    /// The vendors registered in two declare theirs: Anthropic's for
    /// `minimax`, OpenAI's for `zai`, `moonshot`, `xiaomimimo`, `ollama`,
    /// `opencode` and `opencode-go`. A model reference that names no family
    /// is reached by its catalog row's, which is the primary unless the row
    /// says otherwise.
    pub fn catalog(vendor: &str) -> Option<Self> {
        let primary = primary(vendor).and_then(|format| Self::new(vendor, format));
        primary
            .or_else(|| Self::vendor_selections(vendor).next())
            .or_else(|| {
                CATALOG_ONLY
                    .iter()
                    .find(|provider| provider.vendor == vendor)
                    .map(|provider| Self(Kind::CatalogOnly(provider)))
            })
            .or_else(|| declared(vendor).map(|name| Self(Kind::Declared(name))))
    }

    /// Declare `vendor`, a provider this build has no preset for, such as a
    /// company gateway (`corp`) or a local server under a name of its own,
    /// and return the id the catalog files its models under. From then on
    /// [`Self::catalog`] finds it, so `corp/big` resolves, parses and reads
    /// back like a built-in reference once the catalog lists the model
    /// ([`Catalog::insert`]). The registry cannot configure it: whoever
    /// declares it says how its models are reached, such as with
    /// [`Connector::provide`](crate::catalog::Connector::provide) or by
    /// building a [`ProviderConfig`] from a dialect of its own.
    ///
    /// Declaring is idempotent and process-wide. A vendor this build already
    /// knows comes back as [`Self::catalog`] gives it.
    ///
    /// ```
    /// use rig_core::catalog::{Catalog, ModelSpec};
    /// use rig_core::providers::registry::ProviderId;
    ///
    /// let corp = ProviderId::declare("corp")?;
    /// let mut catalog = Catalog::builtin().clone();
    /// catalog.insert(ModelSpec::new(corp, "big").with_tools(true));
    /// assert_eq!(catalog.resolve("corp/big").map_err(|e| e.to_string())?.spec.provider, corp);
    /// assert_eq!(ProviderId::declare("anthropic")?, ProviderId::catalog("anthropic").ok_or("known")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SelectionError::Malformed`] for an empty name, or one with a `/`
    /// or whitespace, which a `vendor/model` reference could not spell.
    pub fn declare(vendor: &str) -> Result<Self, SelectionError> {
        if vendor.is_empty() || vendor.contains('/') || vendor.contains(char::is_whitespace) {
            return Err(SelectionError::Malformed {
                selection: vendor.to_owned(),
            });
        }
        if let Some(known) = Self::catalog(vendor) {
            return Ok(known);
        }
        let mut names = DECLARED
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let name = match names.iter().copied().find(|name| *name == vendor) {
            Some(name) => name,
            None => {
                let name: &'static str = Box::leak(vendor.to_owned().into_boxed_str());
                names.push(name);
                name
            }
        };
        Ok(Self(Kind::Declared(name)))
    }

    /// Every vendor name [`Self::catalog`] knows, registered, catalog-only
    /// or declared. A vendor registered in two families appears twice.
    pub(crate) fn catalog_vendors() -> impl Iterator<Item = &'static str> {
        let declared: Vec<&'static str> = DECLARED
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        Self::all()
            .map(|id| id.vendor())
            .chain(CATALOG_ONLY.iter().map(|provider| provider.vendor))
            .chain(declared)
    }

    /// The vendor, spelled as the provider's own descriptor name.
    pub fn vendor(&self) -> &'static str {
        match &self.0 {
            Kind::Registered(registered) => registered.vendor(),
            Kind::CatalogOnly(provider) => provider.vendor,
            Kind::Declared(vendor) => vendor,
        }
    }

    /// The protocol family, or `None` for a catalog-only or declared
    /// provider.
    pub fn format(&self) -> Option<Format> {
        match &self.0 {
            Kind::Registered(registered) => Some(registered.format()),
            Kind::CatalogOnly(_) | Kind::Declared(_) => None,
        }
    }

    /// Whether the registry can configure this provider. A catalog-only
    /// provider is served by its companion crate instead, and a declared one
    /// by whoever declared it.
    pub fn is_registered(&self) -> bool {
        matches!(self.0, Kind::Registered(_))
    }

    /// Whether this is a vendor declared at run time ([`Self::declare`]).
    pub fn is_declared(&self) -> bool {
        matches!(self.0, Kind::Declared(_))
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
        Self::selection(vendor, family)
    }

    /// Build this selection's preset with `api_key`, or `None` for a
    /// catalog-only or declared provider. Copilot requires an exchanged
    /// session token, not a GitHub OAuth token.
    pub fn config(&self, api_key: impl Into<Secret>) -> Option<ProviderConfig> {
        match &self.0 {
            Kind::Registered(registered) => Some(registered.config(api_key)),
            Kind::CatalogOnly(_) | Kind::Declared(_) => None,
        }
    }

    /// The environment variables tried for the key, in order: the
    /// dialect's own, then its alternative credential's (Azure's
    /// `AZURE_TOKEN`). An empty value counts as unset. Empty for a
    /// catalog-only provider, whose companion crate reads its own
    /// credentials, and for a declared one.
    pub fn api_key_envs(&self) -> Vec<&'static str> {
        match &self.0 {
            Kind::Registered(registered) => registered.key_envs(),
            Kind::CatalogOnly(_) | Kind::Declared(_) => Vec::new(),
        }
    }

    /// Whether this selection needs a credential at all. `false` for a
    /// declared provider: whoever reaches it holds what it needs.
    ///
    /// A local `llama-server` authenticates optionally, so naming its
    /// variable is a hint rather than a requirement.
    pub fn requires_credential(&self) -> bool {
        match &self.0 {
            Kind::Registered(registered) => registered.requires_credential(),
            Kind::CatalogOnly(provider) => provider.credential,
            Kind::Declared(_) => false,
        }
    }

    /// Whether this selection's key is a session token that expires within
    /// the hour, so a key saved for later runs soon stops working.
    pub fn short_lived_key(&self) -> bool {
        matches!(&self.0, Kind::Registered(registered) if registered.short_lived_key())
    }

    /// The registered selection, or [`ConnectError::CatalogOnly`].
    fn registered(self) -> Result<Registered, ConnectError> {
        match self.0 {
            Kind::Registered(registered) => Ok(registered),
            Kind::CatalogOnly(provider) => Err(ConnectError::CatalogOnly {
                vendor: provider.vendor.to_owned(),
                served_by: provider.home,
            }),
            Kind::Declared(vendor) => Err(ConnectError::CatalogOnly {
                vendor: vendor.to_owned(),
                served_by: "whoever declared it: provide it with `Connector::provide`",
            }),
        }
    }

    /// `vendor` in `format`, or why this build registers no such pair.
    fn selection(vendor: &str, format: Format) -> Result<Self, SelectionError> {
        Self::new(vendor, format).ok_or_else(|| {
            let alternatives = alternatives(vendor);
            match alternatives.is_empty() {
                true => SelectionError::Unknown {
                    vendor: vendor.to_owned(),
                },
                false => SelectionError::Unregistered {
                    vendor: vendor.to_owned(),
                    format,
                    alternatives,
                },
            }
        })
    }
}

/// The primary family of `vendor` when it is registered in more than one,
/// as models.dev lists the vendor's own API. [`ProviderId::catalog`] reads it.
fn primary(vendor: &str) -> Option<Format> {
    match vendor {
        "minimax" => Some(Format::Anthropic),
        "zai" | "moonshot" | "xiaomimimo" | "ollama" | "opencode" | "opencode-go" => {
            Some(Format::OpenAi)
        }
        _ => None,
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

/// Reads a selection as [`ProviderId::resolve`] does, and a catalog-only or
/// declared provider by its vendor name, so a catalog id reads back. A
/// declared vendor reads back only once this process declared it.
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

    /// The completion wire for `model`, bound to `http` and encoding and
    /// pricing with `facts`, erased behind a [`ModelAdapter`] labelled
    /// `label`: [`Self::completion_model`] as an effect handler.
    pub fn completion_handler(
        &self,
        label: &str,
        model: &str,
        http: DynHttpClient,
        facts: ModelFacts,
    ) -> ErasedHandler {
        ErasedHandler::new(ModelAdapter::new(
            label,
            self.completion_model(model, http, facts),
        ))
    }

    /// The provider's completion model for `model` on `http`, encoding with
    /// `facts` and pricing its replies by them, erased: how a model the
    /// catalog lists under a declared vendor ([`ProviderId::declare`]) is
    /// reached on a gateway of its own, with its catalog prices. Pass
    /// [`ModelFacts::new`] of the model's spec, or
    /// [`Catalog::facts`](crate::catalog::Catalog::facts).
    ///
    /// The model uses the provider's own completion wire, including
    /// Copilot's model-dependent routing and request envelope. Explicit
    /// hosts, routes and typed options are preserved.
    pub fn completion_model(
        &self,
        model: &str,
        http: DynHttpClient,
        facts: ModelFacts,
    ) -> DynModel<Completion> {
        match self {
            Self::OpenAi(provider) => {
                let wire = openai::wire::OpenAiWire::routed(provider.clone(), model, &facts)
                    .with_facts(facts);
                provider.clone().connect(http).model(wire).erase()
            }
            Self::Anthropic(provider) => {
                let mut model = provider.clone().connect(http).completion(model);
                model.wire = model.wire.with_facts(facts);
                model.erase()
            }
            Self::Gemini(provider) => {
                let mut model = provider.clone().connect(http).completion(model);
                model.wire = model.wire.with_facts(facts);
                model.erase()
            }
        }
    }
}

/// What [`Catalog::connect`] connects to: a catalog entry, or a
/// `vendor/model` reference, as [`Catalog::resolve`] reads one.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ModelSelector<'a> {
    /// A catalog entry: its provider, family and id.
    Spec(&'a ModelSpec),
    /// `vendor/model`, or the older `vendor/format:model`.
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

/// How [`Catalog::connect_with`] connects: an explicit key and the account
/// it belongs to, an HTTP client, a protocol family and a base URL. Each one
/// left unset takes its default: the key from the environment, the shared
/// reqwest client, the family the catalog row is reached by, and the
/// dialect's base URL.
///
/// ```no_run
/// use rig_core::catalog::Catalog;
/// use rig_core::providers::registry::{ConnectOptions, Format};
///
/// let options = ConnectOptions::new().format(Format::Anthropic);
/// let model = Catalog::builtin().connect_with("zai/glm-5", options)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[non_exhaustive]
#[derive(Clone, Default)]
pub struct ConnectOptions {
    api_key: Option<Secret>,
    account_id: Option<String>,
    pub(crate) http: Option<DynHttpClient>,
    format: Option<Format>,
    base_url: Option<String>,
}

impl ConnectOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Credential with `api_key` instead of reading the environment. Only
    /// the key is given: no other variable is read either.
    pub fn api_key(mut self, api_key: impl Into<Secret>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Name `account_id` as the account the key belongs to, as a ChatGPT
    /// subscription credential needs (the `ChatGPT-Account-Id` header).
    /// Only the OpenAI family sends it; the others ignore it.
    ///
    /// ```no_run
    /// use rig_core::catalog::Catalog;
    /// use rig_core::providers::registry::ConnectOptions;
    ///
    /// let options = ConnectOptions::new().api_key("access-token").account_id("account");
    /// let model = Catalog::builtin().connect_with("chatgpt/gpt-5.5", options)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Send through `http` instead of the shared reqwest client.
    pub fn http(mut self, http: impl HttpClientExt + 'static) -> Self {
        self.http = Some(DynHttpClient::new(http));
        self
    }

    /// Speak `format` instead of the family the catalog row is reached by.
    /// The vendor must be registered for it.
    pub fn format(mut self, format: Format) -> Self {
        self.format = Some(format);
        self
    }

    /// Send to `base_url`, such as a proxy, instead of the dialect's.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }
}

/// Never shows the key.
impl fmt::Debug for ConnectOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<set>"))
            .field("account_id", &self.account_id.as_ref().map(|_| "<set>"))
            .field("http", &self.http.as_ref().map(|_| "<set>"))
            .field("format", &self.format)
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Why [`Catalog::connect`] built no model.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    /// The reference is not `vendor/model`.
    #[error("`{reference}` names no model: expected `vendor/model`")]
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
    /// The reference names a vendor this build does not know. Its
    /// suggestions are close references the catalog lists.
    #[error(transparent)]
    NotFound(#[from] NotFound),
    /// The vendor is not registered for the family asked for.
    #[error(transparent)]
    Selection(#[from] SelectionError),
    /// No key was given and none of these variables is set to a non-empty
    /// value.
    #[error(
        "no key for `{vendor}`: set {} or pass one with `ConnectOptions::api_key`",
        tried.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(" or ")
    )]
    MissingKey {
        /// The vendor.
        vendor: String,
        /// The variables read, in order: built-in names, or names a
        /// provider reads at run time.
        tried: Vec<Cow<'static, str>>,
    },
    /// The provider serving the model failed, in its own words, such as a
    /// [`Provide`](crate::catalog::Provide) whose gateway is not configured.
    #[error("{0}")]
    Provider(String),
    /// A variable the dialect reads is set but unusable.
    #[error(transparent)]
    Env(#[from] EnvError),
    /// No HTTP client was given, and this build has no shared one (the
    /// `reqwest` feature is off).
    #[error("no HTTP client: pass one with `ConnectOptions::http`")]
    NoHttpClient,
}

impl Catalog {
    /// The completion model `model` selects, with the key from the
    /// environment, on the shared reqwest client. See
    /// [`Self::connect_with`].
    ///
    /// ```no_run
    /// use rig_core::catalog::Catalog;
    ///
    /// let catalog = Catalog::builtin();
    /// let model = catalog.connect("anthropic/claude-opus-5-5")?; // ANTHROPIC_API_KEY
    /// let spec = catalog.resolve("openai/gpt-5.5")?.spec;
    /// let other = catalog.connect(spec)?; // OPENAI_API_KEY
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn connect<'a>(
        &self,
        model: impl Into<ModelSelector<'a>>,
    ) -> Result<DynModel<Completion>, ConnectError> {
        self.connect_with(model, ConnectOptions::new())
    }

    /// The completion model `model` selects, connected as `options` say.
    ///
    /// A reference is read as [`Self::resolve`] reads it, and the model id
    /// is sent as written. A model this catalog does not list still
    /// connects, on a vendor it knows. The family is `options`'s, else the
    /// one the older `vendor/format:model` spelling names, else the one the
    /// catalog row (or the spec) is reached by, else the vendor's primary.
    /// Unless `options` give a key, it is read from the first of the
    /// vendor's [`api_key_envs`](ProviderId::api_key_envs) set to a
    /// non-empty value, along with the base URL and other settings its
    /// dialect reads from the environment. An account `options` name
    /// replaces the one the dialect reads.
    ///
    /// The model encodes and prices with this catalog's facts: the spec it
    /// was connected from, and this catalog for any other model a request
    /// names, so the catalog's overrides reach its requests and reported
    /// cost.
    ///
    /// # Errors
    ///
    /// [`ConnectError::Malformed`] for a reference that is not
    /// `vendor/model`, [`ConnectError::NotFound`] for an unknown vendor,
    /// [`ConnectError::CatalogOnly`] for a provider a companion crate
    /// serves, [`ConnectError::Selection`] for a family the vendor is not
    /// registered for, [`ConnectError::MissingKey`] when no key is given or
    /// set, and [`ConnectError::NoHttpClient`] without the `reqwest`
    /// feature and an HTTP client.
    pub fn connect_with<'a>(
        &self,
        model: impl Into<ModelSelector<'a>>,
        options: ConnectOptions,
    ) -> Result<DynModel<Completion>, ConnectError> {
        let (provider, model, facts) = match model.into() {
            ModelSelector::Spec(spec) => (
                spec.provider,
                spec.id.as_str(),
                ModelFacts::new(spec.clone()).with_catalog(self),
            ),
            ModelSelector::Reference(text) => {
                let reference = Reference::parse(text).ok_or_else(|| ConnectError::Malformed {
                    reference: text.to_owned(),
                })?;
                let vendor =
                    ProviderId::catalog(reference.vendor).ok_or_else(|| self.not_found(text))?;
                let provider = self
                    .get_vendor(reference.vendor, reference.model)
                    .map_or(vendor, |resolved| resolved.spec.provider);
                let provider = match reference.format {
                    Some(format) if provider.is_registered() => {
                        ProviderId::selection(reference.vendor, format)?
                    }
                    _ => provider,
                };
                (
                    provider,
                    reference.model,
                    self.facts(provider, reference.model),
                )
            }
        };
        let provider = match options.format {
            Some(format) if provider.is_registered() => {
                ProviderId::selection(provider.vendor(), format)?
            }
            _ => provider,
        };
        let registered = provider.registered()?;
        let config = match options.api_key {
            Some(api_key) => registered.config(api_key),
            None => registered.config_from_env()?,
        };
        let config = match (config, options.account_id) {
            (ProviderConfig::OpenAi(provider), Some(account_id)) => {
                ProviderConfig::OpenAi(provider.with_account_id(account_id))
            }
            (config, _) => config,
        };
        let config = match options.base_url {
            Some(base_url) => config.with_base_url(base_url),
            None => config,
        };
        let http = match options.http {
            Some(http) => http,
            None => shared_http()?,
        };
        Ok(config.completion_model(model, http, facts))
    }
}

/// The shared reqwest client.
#[cfg(feature = "reqwest")]
fn shared_http() -> Result<DynHttpClient, ConnectError> {
    Ok(rig_reqwest::shared())
}

/// No shared client without the `reqwest` feature.
#[cfg(not(feature = "reqwest"))]
fn shared_http() -> Result<DynHttpClient, ConnectError> {
    Err(ConnectError::NoHttpClient)
}

#[cfg(test)]
mod tests;
