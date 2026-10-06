//! Typed per-provider request options and reply extras. A provider's
//! [`ProviderExtension`] marker names its key, its serialize-only `Options`
//! and its `Extras` view of a reply. [`ProviderOptions`] holds entries for
//! several providers at once, and each wire reads only the entry named by
//! its own provider.
//!
//! An entry is an object of sections: [`SHARED`] (`"*"`) for fields every
//! route of the provider spells the same, and one section per route, named
//! by its API (`"openai.chat"`, `"openai.responses"`). A wire merges `"*"`,
//! then the section of the route it encodes, at the top level of the body,
//! above the mapped generation options and below `additional_params`. A
//! section for another route is skipped.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//!
//! let request = CompletionRequest::new("hi").provider_options(ProviderOptions::new());
//! assert!(request.provider_options.is_empty());
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::completion::{CompletionRequest, ReplayTarget};
use crate::message::Api;

/// The section every route of a provider reads.
pub const SHARED: &str = "*";

/// A provider's typed request options and reply extras.
pub trait ProviderExtension {
    /// The provider's key: the name its wires report as
    /// [`ReplayTarget::provider`] and stamp on a reply's `Origin`.
    const PROVIDER: &'static str;
    /// The request options: an object of sections, [`SHARED`] and route API
    /// names, each an object of body keys.
    type Options: ExtensionOptions;
    /// The typed view of a reply's `raw` document.
    type Extras: ReplyExtras;
}

/// A provider's request options. Serialize-only: the body keys they write
/// are their only output. They are plain data, `Send + Sync` on every
/// target, since a request carrying them is a component of an ECS world.
pub trait ExtensionOptions: Serialize + Clone + fmt::Debug + Send + Sync + 'static {
    /// The fields `target` cannot send for `request`, each a top-level body
    /// key with the reason. Each one set is reported through the request's
    /// [`OnUnsupported`](crate::completion::OnUnsupported) policy under the
    /// name `"<provider>.<section>.<field>"`. It must not call
    /// [`options::param`](crate::completion::options::param), which reads
    /// it. By default every field is sent.
    fn unsupported(
        &self,
        target: &dyn ReplayTarget,
        request: &CompletionRequest,
    ) -> Vec<(&'static str, String)> {
        let _ = (target, request);
        Vec::new()
    }
}

/// A typed view of a reply's provider document.
pub trait ReplyExtras: Sized {
    /// Read the view from `raw`, the reply's document on `api`.
    ///
    /// # Errors
    ///
    /// When `raw` does not hold the view's shape.
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error>;
}

/// The value at `pointer` in a reply document, read as `T`: `None` when it
/// is absent, `null` or an empty list, which a unary body may state where
/// the document a stream rebuilds omits it. The one reader every `Extras`
/// type's `from_reply` goes through.
///
/// # Errors
///
/// When the value is present but is not a `T`.
pub(crate) fn reply_field<T: serde::de::DeserializeOwned>(
    raw: &Value,
    pointer: &str,
) -> Result<Option<T>, serde_json::Error> {
    match raw.pointer(pointer) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) if items.is_empty() => Ok(None),
        Some(value) => T::deserialize(value).map(Some),
    }
}

/// Why a provider's options cannot be stored.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum OptionsError {
    /// The options failed to serialize.
    #[error("{provider} options do not serialize: {source}")]
    Serialize {
        /// The provider key.
        provider: &'static str,
        /// The serializer's error.
        #[source]
        source: serde_json::Error,
    },
    /// The options are not an object whose every value is an object.
    #[error("{provider} options must serialize to an object of sections, each an object")]
    NotSections {
        /// The provider key.
        provider: &'static str,
    },
}

/// The refusal check of the typed options an entry was made from.
trait Refusals: fmt::Debug + Send + Sync {
    fn refusals(
        &self,
        target: &dyn ReplayTarget,
        request: &CompletionRequest,
    ) -> Vec<(&'static str, String)>;
}

impl<T: ExtensionOptions> Refusals for T {
    fn refusals(
        &self,
        target: &dyn ReplayTarget,
        request: &CompletionRequest,
    ) -> Vec<(&'static str, String)> {
        self.unsupported(target, request)
    }
}

/// One provider's sections, and the typed options they came from.
#[derive(Clone)]
struct Entry {
    sections: Map<String, Value>,
    typed: Option<Arc<dyn Refusals>>,
}

/// Typed options for several providers, one entry per provider key. Each
/// wire reads only the entry named by its own provider. Equality and the
/// serialized form are the entries' sections. A deserialized entry has no
/// typed options behind it, so its fields are sent as written: inserting the
/// typed options again restores their refusal check.
#[derive(Clone, Default)]
pub struct ProviderOptions(BTreeMap<String, Entry>);

impl ProviderOptions {
    /// No entry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `options` as `P`'s entry, replacing any entry it had. Options
    /// that write no field leave `P` with no entry.
    ///
    /// # Errors
    ///
    /// When `options` does not serialize to an object of object sections.
    pub fn insert<P: ProviderExtension>(
        &mut self,
        options: &P::Options,
    ) -> Result<&mut Self, OptionsError> {
        let provider = P::PROVIDER;
        let value = serde_json::to_value(options)
            .map_err(|source| OptionsError::Serialize { provider, source })?;
        let Value::Object(sections) = value else {
            return Err(OptionsError::NotSections { provider });
        };
        let sections =
            non_empty_sections(sections).ok_or(OptionsError::NotSections { provider })?;
        if sections.is_empty() {
            self.0.remove(provider);
        } else {
            self.0.insert(
                provider.to_owned(),
                Entry {
                    sections,
                    typed: Some(Arc::new(options.clone())),
                },
            );
        }
        Ok(self)
    }

    /// `self` with `options` as `P`'s entry. See [`Self::insert`].
    ///
    /// # Errors
    ///
    /// As [`Self::insert`].
    pub fn with<P: ProviderExtension>(
        mut self,
        options: &P::Options,
    ) -> Result<Self, OptionsError> {
        self.insert::<P>(options)?;
        Ok(self)
    }

    /// `P`'s sections as the wire receives them, when it has an entry.
    pub fn get<P: ProviderExtension>(&self) -> Option<&Map<String, Value>> {
        self.0.get(P::PROVIDER).map(|entry| &entry.sections)
    }

    /// Remove `P`'s entry.
    pub fn remove<P: ProviderExtension>(&mut self) {
        self.0.remove(P::PROVIDER);
    }

    /// Whether `P` has an entry.
    pub fn contains<P: ProviderExtension>(&self) -> bool {
        self.0.contains_key(P::PROVIDER)
    }

    /// Whether no provider has an entry.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `self` with each entry of `over` in place of its own for the same
    /// provider. An agent's options overlaid with a run's give the run's
    /// entry for every provider the run names.
    pub fn overlay(mut self, over: &ProviderOptions) -> ProviderOptions {
        for (provider, entry) in &over.0 {
            self.0.insert(provider.clone(), entry.clone());
        }
        self
    }

    /// The sections of `provider`'s entry.
    pub(crate) fn sections(&self, provider: &str) -> Option<&Map<String, Value>> {
        self.0.get(provider).map(|entry| &entry.sections)
    }

    /// The refusals of the typed options behind `provider`'s entry.
    pub(crate) fn refusals(
        &self,
        provider: &str,
        target: &dyn ReplayTarget,
        request: &CompletionRequest,
    ) -> Vec<(&'static str, String)> {
        self.0
            .get(provider)
            .and_then(|entry| entry.typed.as_ref())
            .map(|typed| typed.refusals(target, request))
            .unwrap_or_default()
    }

    /// Remove `field` from the `sections` of `provider`'s entry.
    pub(crate) fn remove_field(&mut self, provider: &str, sections: &[&str], field: &str) {
        if let Some(entry) = self.0.get_mut(provider) {
            for section in sections {
                if let Some(Value::Object(fields)) = entry.sections.get_mut(*section) {
                    fields.shift_remove(field);
                }
            }
        }
    }
}

/// `sections` without its empty sections, or `None` when one is not an
/// object.
fn non_empty_sections(sections: Map<String, Value>) -> Option<Map<String, Value>> {
    let mut kept = Map::new();
    for (name, section) in sections {
        match section {
            Value::Object(fields) if fields.is_empty() => {}
            Value::Object(fields) => {
                kept.insert(name, Value::Object(fields));
            }
            _ => return None,
        }
    }
    Some(kept)
}

impl fmt::Debug for ProviderOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.0
                    .iter()
                    .map(|(provider, entry)| (provider, &entry.sections)),
            )
            .finish()
    }
}

impl PartialEq for ProviderOptions {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|((a, x), (b, y))| a == b && x.sections == y.sections)
    }
}

impl Serialize for ProviderOptions {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(
            self.0
                .iter()
                .map(|(provider, entry)| (provider, &entry.sections)),
        )
    }
}

impl<'de> Deserialize<'de> for ProviderOptions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let entries = BTreeMap::<String, Map<String, Value>>::deserialize(deserializer)?;
        let mut options = BTreeMap::new();
        for (provider, sections) in entries {
            let sections = non_empty_sections(sections).ok_or_else(|| {
                serde::de::Error::custom(format!(
                    "{provider} options must be an object of sections, each an object"
                ))
            })?;
            if !sections.is_empty() {
                options.insert(
                    provider,
                    Entry {
                        sections,
                        typed: None,
                    },
                );
            }
        }
        Ok(Self(options))
    }
}

#[cfg(test)]
mod tests;
